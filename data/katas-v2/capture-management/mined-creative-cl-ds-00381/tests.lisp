;;; tests.lisp — generated from j14i/cl-ds row 381
;;; instruction: Write a Common Lisp macro `lerp` that linearly interpolates between `a` and `b` by ratio `r`, computing a*(1-r) + b*r. T
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((lerp (start) (end) 0.5) . (LET ((#:G1 (START)) (#:G2 (END)) (#:G3 0.5)) (+ (* #:G1 (- 1 #:G3)) (* #:G2 #:G3)))))
