;;; tests.lisp — generated from j14i/cl-ds row 91
;;; instruction: Write a Common Lisp macro `fluid-let` that temporarily assigns new values to a list of special variables for the duratio
;;; category: resource-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((fluid-let ((*debug* t)) (run)) . (LET ((#:G1 *DEBUG*)) (UNWIND-PROTECT (PROGN (SETF *DEBUG* T) (RUN)) (SETF *DEBUG* #:G1)))))
