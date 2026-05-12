;;; tests.lisp — generated from j14i/cl-ds row 227
;;; instruction: Write a Common Lisp macro `with-invariant` that asserts an invariant both before and after evaluating body, returning th
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-invariant (>= balance 0) (deposit 100)) . (PROGN (ASSERT (>= BALANCE 0)) (LET ((#:G1 (PROGN (DEPOSIT 100)))) (ASSERT (>= BALANCE 0)) #:G1))))
