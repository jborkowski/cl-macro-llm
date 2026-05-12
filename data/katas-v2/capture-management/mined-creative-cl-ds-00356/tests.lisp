;;; tests.lisp — generated from j14i/cl-ds row 356
;;; instruction: Write a Common Lisp macro `inc-cube` that increments a place by the cube of an expression while evaluating the expressio
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((inc-cube total (random 5)) . (LET ((#:G1 (RANDOM 5))) (INCF TOTAL (* #:G1 #:G1 #:G1)))))
