;;; tests.lisp — generated from j14i/cl-ds row 38
;;; instruction: Write a Common Lisp macro `clamp` that bounds its first argument between a lower and upper limit. All three subforms mus
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((clamp v 0 255) . (LET ((#:G1 V) (#:G2 0) (#:G3 255)) (COND ((< #:G1 #:G2) #:G2) ((> #:G1 #:G3) #:G3) (T #:G1)))))
