;;; tests.lisp — generated from j14i/cl-ds row 204
;;; instruction: Write a Common Lisp macro `power-of-2-p` that returns true iff its single argument is a positive power of two, evaluatin
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((power-of-2-p (length items)) . (LET ((#:G1 (LENGTH ITEMS))) (AND (PLUSP #:G1) (ZEROP (LOGAND #:G1 (1- #:G1)))))))
