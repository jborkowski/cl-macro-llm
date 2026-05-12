;;; tests.lisp — generated from j14i/cl-ds row 141
;;; instruction: Write a Common Lisp macro `with-saved-vars` that saves the current values of a list of variables into hidden gensyms, ru
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-saved-vars (*a* *b*) (mutate)) . (LET ((#:G1 *A*) (#:G2 *B*)) (UNWIND-PROTECT (PROGN (MUTATE)) (SETF *A* #:G1) (SETF *B* #:G2)))))
