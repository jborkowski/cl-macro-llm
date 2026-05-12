;;; tests.lisp — generated from j14i/cl-ds row 119
;;; instruction: Write a Common Lisp macro `with-saved-stream-position` that records a stream's file position before running body and see
;;; category: resource-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((with-saved-stream-position s (read-line s)) . (LET* ((#:G1 S) (#:G2 (FILE-POSITION #:G1))) (UNWIND-PROTECT (PROGN (READ-LINE S)) (FILE-POSITION #:G1 #:G2)))))
