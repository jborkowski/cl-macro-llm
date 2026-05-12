;;; tests.lisp — generated from j14i/cl-ds row 331
;;; instruction: Write a Common Lisp macro `with-database-transaction` that calls `begin-transaction` on a connection, runs body, and com
;;; category: resource-management | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-database-transaction (db) (insert db row)) . (LET ((#:G1 DB) (#:G2 NIL)) (BEGIN-TRANSACTION #:G1) (UNWIND-PROTECT (MULTIPLE-VALUE-PROG1 (PROGN (INSERT DB ROW)) (SETF #:G2 T)) (IF #:G2 (COMMIT-TRANSACTION #:G1) (ROLLBACK-TRANSACTION #:G1))))))
