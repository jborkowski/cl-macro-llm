;;; tests.lisp — generated from j14i/cl-ds row 383
;;; instruction: Write a Common Lisp macro `mod-into-range` that maps a value into the half-open interval [lo, hi) by computing lo + (val
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((mod-into-range (read-val) 0 100) . (LET* ((#:G1 (READ-VAL)) (#:G2 0) (#:G3 100)) (+ #:G2 (MOD (- #:G1 #:G2) (- #:G3 #:G2))))))
