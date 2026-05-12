;;; tests.lisp — generated from j14i/cl-ds row 72
;;; instruction: Write a Common Lisp macro `assert-all` that evaluates its first argument once and asserts that every supplied unary pred
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-all x numberp plusp) . (LET ((#:G1 X)) (ASSERT (NUMBERP #:G1)) (ASSERT (PLUSP #:G1)) #:G1)))
