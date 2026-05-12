;;; tests.lisp — generated from j14i/cl-ds row 485
;;; instruction: Write a Common Lisp macro `with-temp-package` that creates a fresh package with a unique name, binds it to a variable fo
;;; category: resource-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-temp-package (p) (intern "FOO" p)) . (LET* ((#:G1 (FORMAT NIL "TMP-~A" (GENSYM))) (P (MAKE-PACKAGE #:G1 :USE '(:CL)))) (UNWIND-PROTECT (PROGN (INTERN "FOO" P)) (DELETE-PACKAGE P)))))
