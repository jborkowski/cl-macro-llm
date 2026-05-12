;;; setup.lisp — alambda, anaphoric lambda (Graham, On Lisp; Hoyte, LoL).
;;; KEY HYGIENE NOTE: `self` is INTENTIONALLY captured via LABELS so the
;;; lambda body can recurse on itself by name. Do NOT gensym `self`.
;;; This is the classic anaphoric self-reference pattern — without the
;;; capture, recursion would have to thread the function explicitly.

(in-package :cl-user)
