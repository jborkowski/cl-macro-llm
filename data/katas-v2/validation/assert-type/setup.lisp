;;; setup.lisp — assert-type, signal a TYPE-ERROR if a place's value
;;; doesn't match the named type. Idiomatic CL: prefer signalling the
;;; condition explicitly so callers can HANDLER-CASE it.

(in-package :cl-user)
