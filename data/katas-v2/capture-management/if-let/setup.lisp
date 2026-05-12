;;; setup.lisp — if-let, caller-named conditional binding (Alexandria).
;;; KEY HYGIENE NOTE: the caller's variable name is captured into the
;;; LET binding so BOTH the then-branch AND the else-branch can refer
;;; to it. Useful when the else-branch wants to inspect the falsy value.

(in-package :cl-user)
