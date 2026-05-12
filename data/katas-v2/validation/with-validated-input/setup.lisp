;;; setup.lisp — with-validated-input. Bind a value, validate it via a
;;; predicate, signal an error on failure, then run the body knowing
;;; the binding satisfies the predicate.

(in-package :cl-user)
