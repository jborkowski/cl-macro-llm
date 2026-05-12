;;; setup.lisp — dbg, Rust-style print-and-return. Evaluate a form,
;;; print "form = value" to *trace-output*, and return the value
;;; unchanged so the macro is transparently insertable into any
;;; expression.

(in-package :cl-user)
