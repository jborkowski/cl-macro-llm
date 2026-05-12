;;; tests.lisp — generated from j14i/cl-ds row 382
;;; instruction: Write a Common Lisp macro `with-stashed-binding` that saves the current value of a variable in a gensym, sets the variab
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-stashed-binding *debug* t (run)) . (LET ((#:G1 *DEBUG*)) (UNWIND-PROTECT (PROGN (SETF *DEBUG* T) (RUN)) (SETF *DEBUG* #:G1)))))
