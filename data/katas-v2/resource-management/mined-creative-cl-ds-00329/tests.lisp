;;; tests.lisp — generated from j14i/cl-ds row 329
;;; instruction: Write a Common Lisp macro `with-pinned-buffer` that calls `pin-buffer` on a buffer, executes body, and guarantees `unpin
;;; category: resource-management | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((with-pinned-buffer (b) (process b)) . (LET ((#:G1 B)) (PIN-BUFFER #:G1) (UNWIND-PROTECT (PROGN (PROCESS B)) (UNPIN-BUFFER #:G1)))))
