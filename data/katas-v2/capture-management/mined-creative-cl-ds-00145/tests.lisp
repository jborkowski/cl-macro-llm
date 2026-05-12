;;; tests.lisp — generated from j14i/cl-ds row 145
;;; instruction: Write a Common Lisp macro `square!` that squares a place in place by capturing the place's current value once in a hidde
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((square! x) . (LET ((#:G1 X)) (SETF X (* #:G1 #:G1)))))
