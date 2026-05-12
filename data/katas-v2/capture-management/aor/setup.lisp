;;; setup.lisp — aor, anaphoric or (Paul Graham, On Lisp variant).
;;; KEY HYGIENE NOTE: `it` is INTENTIONALLY captured so the default
;;; branch can reference the falsy-but-not-nil value (e.g. zero, empty
;;; string) the primary form returned.

(in-package :cl-user)
