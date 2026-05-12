;;; tests.lisp — generated from j14i/cl-ds row 333
;;; instruction: Write a Common Lisp macro `with-stream-position` that records a stream's current `file-position`, runs the body, and see
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-stream-position (fs) (read-line fs)) . (LET* ((#:G1 FS) (#:G2 (FILE-POSITION #:G1))) (UNWIND-PROTECT (PROGN (READ-LINE FS)) (FILE-POSITION #:G1 #:G2)))))
