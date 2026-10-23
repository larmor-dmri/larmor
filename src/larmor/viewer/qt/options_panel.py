"""The Options panel, with a field per option of what is selected in Layers."""

from collections.abc import Callable, Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.options import (
    DTI_UNUSED,
    ItemOptions,
    OptionError,
    OptionGroup,
    OptionInfo,
    OptionName,
    option_fields,
)
from larmor.viewer.model.registry import REGISTRY
from larmor.viewer.model.surface import SurfaceData
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.field import (
    ItemField,
    OptionField,
    SliderField,
    add_field_row,
    field_for,
)
from larmor.viewer.qt.style import form_layout, icon_path

# Makes an item's extra widget, such as a button, shown below a group's fields.
type Extra = Callable[[ItemId, QWidget], QWidget]

# The groups in the order the panel shows them.
GROUP_ORDER: dict[OptionGroup, int] = {g: i for i, g in enumerate(OptionGroup)}


class OptionsPanel(QScrollArea):
    """Edits the selected item's or category's options, in groups that fold away.

    `extras` gives the items of a category a widget at the end of a group, made
    each time the panel shows one of them, and folded with the group.
    """

    def __init__(
        self,
        viewer: Viewer,
        parent: QWidget | None = None,
        extras: Mapping[Category, tuple[OptionGroup, Extra]] | None = None,
    ) -> None:
        """Creates the panel."""
        super().__init__(parent)
        self.viewer = viewer
        self.extras = dict(extras or {})
        self.setObjectName("options")
        self.setWidgetResizable(True)
        self._fields: dict[OptionName, OptionField] = {}
        self._errors: dict[OptionName, QLabel] = {}
        self._headers: dict[OptionGroup, QToolButton] = {}
        self._members: dict[OptionGroup, list[OptionName]] = {}
        self._extra: tuple[OptionGroup, QWidget] | None = None
        # The groups folded away, per category, while the program runs.
        self._folded: set[tuple[Category, OptionGroup]] = set()
        # The viewer can outlive the window: stop listening when the widget goes.
        unsubscribe = self.viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self._build()

    def _on_event(self, event: Event) -> None:
        """Rebuilds the page on a new selection, else refreshes the fields changed."""
        if isinstance(event, ViewStateChanged) and event.name == ViewField.SELECTED:
            self._build()
            return
        target = self.viewer.scene.view.selected
        if target is None:
            return
        # Refresh only the fields an event changes, keeping other text being typed.
        if isinstance(event, OptionChanged):
            if target in (event.item.id, event.item.category):
                self._option_changed(event.name)
        elif isinstance(event, (ItemAdded, ItemRemoved)):
            if target == event.item.category:  # can change a mixed value
                self._update(list(self._fields))
            self._refresh_items()  # the items an option can name
        elif isinstance(event, DataChanged):
            if target in (event.item.id, event.item.category):
                # The sliders span the new data's range.
                self._rebound([n for n, f in self._fields.items() if _slider(f)])
            self._refresh_items()  # the new data can fit other options, or not
        elif (
            isinstance(event, ViewStateChanged)
            and event.name == ViewField.MAIN_VOLUME
            and target == Category.VOLUMES
        ):
            # The Volumes category's Visible follows the main volume.
            self._update([OptionName.VISIBLE])

    def _option_changed(self, name: OptionName) -> None:
        """Shows an option's new value of the item or category shown."""
        if name in (OptionName.VALUES_FILE, OptionName.VALUES_VOLUME):
            # New values to colour by span another range.
            self._rebound([OptionName.MINIMUM, OptionName.MAXIMUM])
        self._update([name])
        if name == OptionName.DTI:
            self._enable()

    def _build(self) -> None:
        """Builds the page of the item or category selected in Layers."""
        form = QWidget()
        form.setObjectName("optionsForm")
        layout = form_layout(form)
        self._fields = {}
        self._errors = {}
        self._headers = {}
        self._members = {}
        self._extra = None
        target = self.viewer.scene.view.selected
        if target is None:
            layout.addRow(QLabel("Select an item or a category in Layers."))
        else:
            self._build_page(layout, form, target)

        self.setWidget(form)
        self._refresh_items()
        self._update(list(self._fields))
        self._enable()

    def _build_page(
        self, layout: QFormLayout, form: QWidget, target: ItemId | Category
    ) -> None:
        """Adds the title and the target's options, by group, to the form."""
        title = QLabel(self._title(target))
        title.setObjectName("formTitle")
        layout.addRow(title)
        category = self._category(target)
        shown = [
            (name, f, info)
            for name, (f, info) in option_fields(self._options_class(target)).items()
            if self._shows(target, name, info)
        ]
        # Sorted by group only, so each group keeps its fields' declaration order.
        shown.sort(key=lambda s: GROUP_ORDER[_group(s[2])])
        ends: dict[OptionGroup, int] = {}  # the row after each group's last
        for name, f, info in shown:
            group = _group(info)
            if group not in self._headers:
                self._headers[group] = self._header(layout, category, group, form)
                self._members[group] = []
            field = field_for(f, info, form, self._bounds(target, name))
            error = QLabel(form)
            error.setObjectName("fieldError")
            error.setWordWrap(True)
            error.hide()

            field.edited.connect(lambda value, name=name: self._edit(name, value))
            add_field_row(layout, info.label, info.help, field, form)
            layout.addRow(error)  # a hidden row takes no room
            self._fields[name] = field
            self._errors[name] = error
            self._members[group].append(name)
            ends[group] = layout.rowCount()
        if not isinstance(target, Category) and category in self.extras:
            group, extra = self.extras[category]
            widget = extra(target, form)
            layout.insertRow(ends[group], widget)
            self._extra = (group, widget)
        for group in self._headers:
            self._show_group(layout, group, (category, group) not in self._folded)

    def _shows(
        self, target: ItemId | Category, name: OptionName, info: OptionInfo
    ) -> bool:
        """Returns whether the option has a field on the target's page."""
        if not info.panel or (info.per_item and isinstance(target, Category)):
            return False
        if name == OptionName.VOLUME_INDEX and self._bounds(target, name)[1] == 0:
            return False  # a 3D image has only volume 0
        if name in (OptionName.DTI, OptionName.MODULATE):
            # Only an image of 3 volumes can be a DTI colour map.
            return _three_volumes(self.viewer, target)
        if name == OptionName.HIDE_MEDIAL_WALL and not isinstance(target, Category):
            # Only a surface with a cortex label has a medial wall to hide.
            data = self.viewer.scene.item(target).data
            return isinstance(data, SurfaceData) and data.cortex is not None
        return True

    def _header(
        self,
        layout: QFormLayout,
        category: Category,
        group: OptionGroup,
        form: QWidget,
    ) -> QToolButton:
        """Adds a group's title, a button that folds the group's fields away."""
        header = QToolButton(form)
        header.setObjectName("optionGroup")
        header.setText(group.value)
        header.setCheckable(True)
        header.setAutoRaise(True)
        header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.toggled.connect(
            lambda shown: self._toggle(layout, category, group, shown)
        )
        layout.addRow(header)
        return header

    def _toggle(
        self, layout: QFormLayout, category: Category, group: OptionGroup, shown: bool
    ) -> None:
        """Folds or unfolds a group, and remembers it for the category."""
        if shown:
            self._folded.discard((category, group))
        else:
            self._folded.add((category, group))
        self._show_group(layout, group, shown)

    def _show_group(self, layout: QFormLayout, group: OptionGroup, shown: bool) -> None:
        """Shows or hides a group's fields; the header shows which."""
        header = self._headers[group]
        header.blockSignals(True)  # no _toggle back
        header.setChecked(shown)
        header.blockSignals(False)
        header.setIcon(QIcon(icon_path("chevron-down" if shown else "chevron-right")))
        for name in self._members[group]:
            layout.setRowVisible(self._fields[name], shown)
            if not shown:
                self._errors[name].hide()  # a folded field's message is dropped
        if self._extra is not None and self._extra[0] == group:
            layout.setRowVisible(self._extra[1], shown)

    def _refresh_items(self) -> None:
        """Gives each field that names an item, or its data, the names it can take."""
        target = self.viewer.scene.view.selected
        for name, field in self._fields.items():
            if isinstance(field, ItemField) and isinstance(target, int):
                if option_fields(self._options_class(target))[name][1].data_key:
                    keys = self.viewer.data_choices(target, name)
                    field.set_choices([(k, k) for k in keys])
                else:
                    choices = self.viewer.item_choices(target, name)
                    field.set_choices([(i.id, i.name) for i in choices])
                field.show_value(self._value(name))

    def _enable(self) -> None:
        """Greys out the fields that DTI colours make unused, or that need them."""
        target = self.viewer.scene.view.selected
        if not isinstance(target, int):
            return  # the per-item DTI option is not on a category's page
        dti = getattr(self.viewer.scene.item(target).options, OptionName.DTI, False)
        for name, field in self._fields.items():
            if name == OptionName.MODULATE:
                field.setEnabled(dti)
            elif name in DTI_UNUSED:
                field.setEnabled(not dti)

    def _bounds(
        self, target: ItemId | Category, name: OptionName
    ) -> tuple[float, float]:
        """Returns the range the data of the item, or of the category's items, spans."""
        if name is OptionName.VOLUME_INDEX:
            assert not isinstance(target, Category)
            item = self.viewer.scene.item(target)
            vol_count = item.volume_count
            assert isinstance(vol_count, int)
            return (0, vol_count - 1)
        scene = self.viewer.scene
        items = (
            scene.in_category(target)
            if isinstance(target, Category)
            else [scene.item(target)]
        )
        ranges = [i.value_range for i in items if i.value_range is not None]
        if not ranges:
            return (0.0, 1.0)  # no data, as in an empty category: the defaults' range
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    def _rebound(self, names: list[OptionName]) -> None:
        """Spans the sliders of these options over the data's range again."""
        target = self.viewer.scene.view.selected
        assert target is not None
        for name in names:
            field = self._fields.get(name)
            if isinstance(field, SliderField):
                field.set_bounds(self._bounds(target, name))
                field.show_value(self._value(name))

    def _title(self, target: ItemId | Category) -> str:
        """Returns the page's title, the item's or category's name."""
        if isinstance(target, Category):
            return REGISTRY[target].label
        return self.viewer.scene.item(target).name

    def _category(self, target: ItemId | Category) -> Category:
        """Returns the target's category."""
        if isinstance(target, Category):
            return target
        return self.viewer.scene.item(target).category

    def _options_class(self, target: ItemId | Category) -> type[ItemOptions]:
        """Returns the options class of the target's items."""
        if isinstance(target, Category):
            return REGISTRY[target].options
        return type(self.viewer.scene.item(target).options)

    def _value(self, name: OptionName) -> object:
        """Returns an option's value for the selection, `MIXED` if items differ."""
        target = self.viewer.scene.view.selected
        assert target is not None  # fields exist only while something is selected
        if isinstance(target, Category):
            return self.viewer._category_value(target, name)
        return getattr(self.viewer.scene.item(target).options, name)

    def _update(self, names: list[OptionName]) -> None:
        """Shows the options' values in their fields and hides their errors."""
        for name in names:
            if name not in self._fields:
                continue  # not shown on this page, as an index on a category's
            self._fields[name].show_value(self._value(name))
            self._errors[name].hide()  # the message was about text now replaced

    def _edit(self, name: OptionName, value: object) -> None:
        """Sets an option from its field; shows the error if it is refused."""
        target = self.viewer.scene.view.selected
        assert target is not None
        try:
            self.viewer.set(target, name, value)
        except OptionError as error:
            self._errors[name].setText(str(error))
            self._errors[name].show()
            return
        self._errors[name].hide()
        # Setting the current value sends no event, so tidy the field here.
        self._fields[name].show_value(self._value(name))


def _group(info: OptionInfo) -> OptionGroup:
    """Returns an item option's group.

    Raises:
        ValueError: If the option has none, as settings do.
    """
    if info.group is None:
        raise ValueError(f"The option {info.label!r} has no group.")
    return info.group


def _slider(field: OptionField) -> bool:
    """Returns whether a field is a slider."""
    return isinstance(field, SliderField)


def _three_volumes(viewer: Viewer, target: ItemId | Category) -> bool:
    """Returns whether the target is an item holding 3 volumes."""
    return (
        not isinstance(target, Category) and viewer.scene.item(target).volume_count == 3
    )
