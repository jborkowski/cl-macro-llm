;;; setup.lisp — nlet, named let (Doug Hoyte, Let Over Lambda ch. 5).
;;; KEY HYGIENE NOTE: the loop name supplied by the caller is captured
;;; into a LABELS binding so the body can recurse by name. The body's
;;; reference to the name MUST resolve to the labels form, so do NOT
;;; gensym the loop name — it is intentionally caller-visible.
;;; This is the canonical port of Scheme's named-let into Common Lisp.

(in-package :cl-user)
