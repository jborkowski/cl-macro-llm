;;; tests.lisp — generated from j14i/cl-ds row 258
;;; instruction: Write a Common Lisp macro `with-collected-output` that captures everything written to *standard-output* during body into
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-collected-output (princ "hi") (princ "!")) . (LET ((#:G1 (MAKE-STRING-OUTPUT-STREAM))) (LET ((*STANDARD-OUTPUT* #:G1)) (PRINC "hi") (PRINC "!")) (GET-OUTPUT-STREAM-STRING #:G1))))
