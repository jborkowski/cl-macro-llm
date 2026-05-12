;;; setup.lisp — with-lock-held, acquire/release pattern around body
;;; via UNWIND-PROTECT. The lock-form is re-evaluated in the cleanup
;;; branch; for a lock-acquisition where the lock object is a stable
;;; variable reference, this is safe and idiomatic.

(in-package :cl-user)
