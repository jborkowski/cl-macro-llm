;;; setup.lisp — check-args, a lightweight precondition guard.
;;; Use at the top of a function body to assert invariants without
;;; the heavyweight machinery of CHECK-TYPE or full contracts.

(in-package :cl-user)
