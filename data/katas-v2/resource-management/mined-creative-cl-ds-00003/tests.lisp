;;; tests.lisp — generated from j14i/cl-ds row 3
;;; instruction: Write a Common Lisp macro `with-cwd` that temporarily sets `*default-pathname-defaults*` to a new pathname, executes the
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-cwd #p"/tmp/" (load "foo.lisp")) . (LET ((#:G1 *DEFAULT-PATHNAME-DEFAULTS*) (#:G2 #P"/tmp/")) (UNWIND-PROTECT (PROGN (SETF *DEFAULT-PATHNAME-DEFAULTS* #:G2) (LOAD "foo.lisp")) (SETF *DEFAULT-PATHNAME-DEFAULTS* #:G1)))))
