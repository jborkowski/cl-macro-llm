;;; tests.lisp — hand-crafted, On Lisp ch. 8 accumulator pattern
;;; instruction: Write `with-collectors` — bind each name as both a
;;; variable AND a flet function that pushes onto it. Return all lists.
;;; category: capture-management | technique: function-capture | complexity: advanced
;;; quality_score: 0.95

(
 ((with-collectors (evens) (evens 2) (evens 4))
  . (LET ((EVENS NIL))
      (FLET ((EVENS (X) (PUSH X EVENS)))
        (EVENS 2)
        (EVENS 4)
        (VALUES (NREVERSE EVENS)))))

 ((with-collectors (lefts rights) (lefts :a) (rights :b) (lefts :c))
  . (LET ((LEFTS NIL) (RIGHTS NIL))
      (FLET ((LEFTS (X) (PUSH X LEFTS))
             (RIGHTS (X) (PUSH X RIGHTS)))
        (LEFTS :A)
        (RIGHTS :B)
        (LEFTS :C)
        (VALUES (NREVERSE LEFTS) (NREVERSE RIGHTS)))))
)
