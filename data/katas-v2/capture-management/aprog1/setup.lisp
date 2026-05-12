;;; setup.lisp — aprog1, anaphoric prog1 (Paul Graham, On Lisp ch. 14).
;;; KEY HYGIENE NOTE: `it` is INTENTIONALLY captured. The body runs side
;;; effects on/around `it`, and the macro returns `it`. Do NOT gensym.

(in-package :cl-user)
