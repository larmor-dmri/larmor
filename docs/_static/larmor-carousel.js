// Advances the landing page's carousel every 5 seconds, with Bootstrap's carousel.
// It pauses while the pointer is over it or it has the keyboard focus, and does not
// advance on its own when the reader prefers reduced motion. The arrow keys, the
// previous and next buttons and the indicators work in every case.
(function () {
  "use strict";
  var INTERVAL_MS = 5000;

  function start() {
    var element = document.getElementById("lar-carousel");
    if (!element || !window.bootstrap || !window.bootstrap.Carousel) {
      return;
    }
    var carousel = window.bootstrap.Carousel.getOrCreateInstance(element, {
      interval: false,
      ride: false,
      pause: false,
      keyboard: true,
      touch: true,
      wrap: true,
    });
    element.classList.add("lar-ready");

    var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    var hovered = false;
    var focused = false;
    element.addEventListener("mouseenter", function () { hovered = true; });
    element.addEventListener("mouseleave", function () { hovered = false; });
    element.addEventListener("focusin", function () { focused = true; });
    element.addEventListener("focusout", function (event) {
      focused = element.contains(event.relatedTarget);
    });

    window.setInterval(function () {
      if (!motion.matches && !hovered && !focused && !document.hidden) {
        carousel.next();
      }
    }, INTERVAL_MS);
  }

  // The theme's bootstrap.js is deferred too, after this script: wait for it.
  if (document.readyState === "complete") {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
})();
