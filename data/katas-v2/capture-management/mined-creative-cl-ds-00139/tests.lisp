;;; tests.lisp — generated from j14i/cl-ds row 139
;;; instruction: Write a Common Lisp macro `with-temp-binding` that saves a place to a hidden gensym, sets it to a new value, runs body, 
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-temp-binding (*x* 10) (do-stuff)) . (LET ((#:G1 *X*)) (UNWIND-PROTECT (PROGN (SETF *X* 10) (DO-STUFF)) (SETF *X* #:G1)))))
