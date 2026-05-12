;;; setup.lisp — awhen, anaphoric when (Paul Graham, On Lisp ch. 14).
;;; KEY HYGIENE NOTE: `it` is INTENTIONALLY captured. The body of the
;;; macro relies on the symbol IT being visible. Do NOT gensym `it`.
;;; This is one of the canonical cases of legitimate variable capture.

(in-package :cl-user)
