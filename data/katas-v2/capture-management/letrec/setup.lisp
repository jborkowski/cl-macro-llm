;;; setup.lisp — letrec, Scheme-style mutually recursive bindings.
;;; KEY HYGIENE NOTE: every binding name is captured as both a LABELS
;;; function and a referenceable name in every other binding's body —
;;; mutual recursion only works if the names stay the caller's literals.
;;; Do NOT gensym. CL's stdlib has no letrec; this is the canonical port.

(in-package :cl-user)
