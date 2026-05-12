;;; setup.lisp — trace-when, run body and emit a trace line, but only
;;; when a runtime condition is true. Cheaper than always-on logging.

(in-package :cl-user)
