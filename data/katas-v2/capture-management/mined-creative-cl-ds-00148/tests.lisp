;;; tests.lisp — generated from j14i/cl-ds row 148
;;; instruction: Write a Common Lisp macro `with-escape` that wraps body in a hidden named block and exposes a user-named local function 
;;; category: capture-management | technique: closure-capture | complexity: intermediate
;;; quality_score: 1.0

(((with-escape (bail) (when (bad?) (bail :err)) (compute)) . (BLOCK #:G1 (FLET ((BAIL (&OPTIONAL #:G2) (RETURN-FROM #:G1 #:G2))) (WHEN (BAD?) (BAIL :ERR)) (COMPUTE)))))
