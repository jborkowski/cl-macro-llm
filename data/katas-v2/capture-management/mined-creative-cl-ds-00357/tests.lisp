;;; tests.lisp — generated from j14i/cl-ds row 357
;;; instruction: Write a Common Lisp macro `bounded-incf` that adds a delta to a place but caps the result at a maximum, evaluating both 
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((bounded-incf counter (random 10) 100) . (LET ((#:G1 (RANDOM 10)) (#:G2 100)) (SETF COUNTER (MIN (+ COUNTER #:G1) #:G2)))))
