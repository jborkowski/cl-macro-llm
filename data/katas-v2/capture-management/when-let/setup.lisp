;;; setup.lisp — when-let, caller-named binding (Alexandria-style).
;;; KEY HYGIENE NOTE: the variable name is supplied BY THE CALLER and
;;; gets captured into the LET binding so the body can reference it.
;;; Unlike `awhen`, no fixed `it` symbol — caller picks the name. This
;;; is still capture-management: the macro must place the caller's
;;; symbol into the binding, NOT gensym it away.

(in-package :cl-user)
