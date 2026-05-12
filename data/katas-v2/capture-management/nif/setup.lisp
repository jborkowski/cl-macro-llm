;;; setup.lisp — nif, numeric-if (Paul Graham, On Lisp ch. 14).
;;; KEY HYGIENE NOTE: `it` is INTENTIONALLY captured so all three
;;; branches (positive / zero / negative) can reference the value
;;; that was classified. Do NOT gensym `it`.

(in-package :cl-user)
