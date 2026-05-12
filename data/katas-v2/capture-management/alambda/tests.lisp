;;; tests.lisp — hand-crafted from On Lisp ch. 14 / Let Over Lambda ch. 5
;;; instruction: Write `alambda` — an anaphoric lambda that captures
;;; the function itself as `self`, enabling recursion without naming.
;;; category: capture-management | technique: anaphoric | complexity: intermediate
;;; quality_score: 0.95

(
 ((alambda (n) (if (= n 0) 1 (* n (self (1- n)))))
  . (LABELS ((SELF (N) (IF (= N 0) 1 (* N (SELF (1- N)))))) #'SELF))

 ((alambda (xs) (if (null xs) 0 (1+ (self (cdr xs)))))
  . (LABELS ((SELF (XS) (IF (NULL XS) 0 (1+ (SELF (CDR XS)))))) #'SELF))

 ((alambda (a b) (if (zerop b) a (self b (mod a b))))
  . (LABELS ((SELF (A B) (IF (ZEROP B) A (SELF B (MOD A B))))) #'SELF))
)
