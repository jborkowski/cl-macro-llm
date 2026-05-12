;;; tests.lisp — generated from j14i/cl-ds row 237
;;; instruction: Write a Common Lisp macro `inline-test` that compares an expected value against an actual expression with EQUAL and prin
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((inline-test "addition" 3 (+ 1 2)) . (LET ((#:G1 3) (#:G2 (+ 1 2))) (IF (EQUAL #:G1 #:G2) (FORMAT *TRACE-OUTPUT* "~&PASS ~A~%" "addition") (FORMAT *TRACE-OUTPUT* "~&FAIL ~A: expected ~S got ~S~%" "addition" #:G1 #:G2)))))
