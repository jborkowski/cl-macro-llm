;;; setup.lisp — run-if-unset (NEGATIVE-condition edition).
;;; ADVERSARIAL PAIR NOTE: this is the UNLESS twin. Its sibling kata
;;; `run-if-set` runs the body when the test is non-nil (uses WHEN).
;;; Reading carelessly will pick the wrong macro.

(in-package :cl-user)
