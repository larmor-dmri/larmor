"""The undo and redo history of a phantom's bundles."""

from dataclasses import dataclass, replace
from typing import Final

from larmor.viewer.model.item import ItemId
from larmor.viewer.model.phantom import BundleSettings, Draft, PhantomState

# The most steps of a phantom's history that are kept.
HISTORY_LIMIT: Final = 100


@dataclass(frozen=True, eq=False)
class BundleSnapshot:
    """The phantom's bundles at one moment, the one being drawn among them.

    Attributes:
        bundles: The finished bundles.
        draft: The bundle being drawn; None if there is none, or the Pen only waits
            for a stroke.
    """

    bundles: tuple[BundleSettings, ...]
    draft: Draft | None

    @classmethod
    def of(cls, state: PhantomState) -> BundleSnapshot:
        """Returns the snapshot of a phantom as it is."""
        draft = state.draft
        return cls(state.bundles, None if draft is None or draft.is_empty else draft)

    def content(self) -> tuple[object, ...]:
        """Returns what the snapshot makes, for telling whether anything changed.

        A bundle being edited counts as what it would become, so that opening and
        closing an edit that changes nothing is no step.
        """
        draft = self.draft
        if draft is None:
            return (self.bundles, None)
        shape = replace(draft.geometry, control_points=draft.points)
        if draft.editing is None:
            return (self.bundles, (draft.points, shape, draft.tissue, draft.branch))
        edited = tuple(
            replace(b, geometry=shape, tissue=draft.tissue, branch=draft.branch)
            if b.item_id == draft.editing.item_id
            else b
            for b in self.bundles
        )
        return (edited, None)

    def item_ids(self) -> set[ItemId]:
        """Returns the tractogram items that the snapshot names."""
        ids = {b.item_id for b in self.bundles}
        if self.draft is not None and self.draft.item_id is not None:
            ids.add(self.draft.item_id)
        return ids


class BundleHistory:
    """Holds the snapshots to go back and forward to, at most `limit` of each.

    Attributes:
        current: The snapshot of the phantom as it stands; None before the first.
    """

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        """Creates an empty history."""
        self.limit = limit
        self.current: BundleSnapshot | None = None
        self._back: list[BundleSnapshot] = []
        self._forward: list[BundleSnapshot] = []

    def reset(self, snapshot: BundleSnapshot | None) -> None:
        """Forgets all steps, and starts again from a snapshot."""
        self.current = snapshot
        self._back.clear()
        self._forward.clear()

    def note(self, snapshot: BundleSnapshot, *, merge: bool = False) -> None:
        """Takes in the phantom as it now stands, as a new step if anything changed.

        Args:
            snapshot: The phantom now.
            merge: Whether this belongs with the last step, as finishing a bundle
                belongs with the stroke before it: going back then skips both.
        """
        if self.current is None:
            self.current = snapshot
        elif merge:
            self.current = snapshot
            self._forward.clear()
        elif snapshot.content() == self.current.content():
            self.current = snapshot
        else:
            self._back.append(self.current)
            del self._back[: -self.limit]
            self._forward.clear()
            self.current = snapshot

    def back(self) -> BundleSnapshot | None:
        """Goes a step back, and returns the snapshot there; None if none is."""
        if not self._back or self.current is None:
            return None
        self._forward.append(self.current)
        self.current = self._back.pop()
        return self.current

    def forward(self) -> BundleSnapshot | None:
        """Goes a step forward, and returns the snapshot there; None if none is."""
        if not self._forward or self.current is None:
            return None
        self._back.append(self.current)
        self.current = self._forward.pop()
        return self.current

    def item_ids(self) -> set[ItemId]:
        """Returns the tractogram items that any snapshot held here names."""
        held = [*self._back, *self._forward]
        if self.current is not None:
            held.append(self.current)
        return {item_id for snapshot in held for item_id in snapshot.item_ids()}
