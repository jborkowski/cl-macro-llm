;;; tests.lisp — generated from j14i/cl-ds row 361
;;; instruction: Write a Common Lisp macro `with-swapped` that swaps the values of two places, runs body, and restores both places to the
;;; category: capture-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-swapped a b (process)) . (LET ((#:G1 A) (#:G2 B)) (SETF A #:G2 B #:G1) (UNWIND-PROTECT (PROGN (PROCESS)) (SETF A #:G1 B #:G2)))))
