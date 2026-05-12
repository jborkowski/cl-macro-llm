;;; tests.lisp — generated from j14i/cl-ds row 7
;;; instruction: Write a Common Lisp macro `with-restored-special` that captures the current value of a special variable, runs the body (
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-restored-special *debug-level* (setf *debug-level* 3) (debug-stuff)) . (LET ((#:G1 *DEBUG-LEVEL*)) (UNWIND-PROTECT (PROGN (SETF *DEBUG-LEVEL* 3) (DEBUG-STUFF)) (SETF *DEBUG-LEVEL* #:G1)))))
