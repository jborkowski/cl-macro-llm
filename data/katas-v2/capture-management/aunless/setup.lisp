;;; setup.lisp — aunless, anaphoric unless (Paul Graham, On Lisp ch. 14).
;;; KEY HYGIENE NOTE: `it` is INTENTIONALLY captured so the else-branch
;;; body can inspect what the test returned (typically nil — for logging
;;; the missing value, distinguishing NIL from no-such-key, etc).

(in-package :cl-user)
