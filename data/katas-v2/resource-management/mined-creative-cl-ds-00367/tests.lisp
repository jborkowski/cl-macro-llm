;;; tests.lisp — generated from j14i/cl-ds row 367
;;; instruction: Write a Common Lisp macro `with-deadline-check` that runs the body and emits a warning if execution exceeded the given n
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-deadline-check 5 (long-computation)) . (LET* ((#:G1 (GET-INTERNAL-REAL-TIME)) (#:G2 (* 5 INTERNAL-TIME-UNITS-PER-SECOND))) (MULTIPLE-VALUE-PROG1 (PROGN (LONG-COMPUTATION)) (WHEN (> (- (GET-INTERNAL-REAL-TIME) #:G1) #:G2) (WARN "Deadline exceeded"))))))
