from operator import attrgetter

from devito.finite_differences.differentiable import DifferentiableOp
from devito.symbolics import retrieve_indexed, split_affine
from devito.tools import PartialOrderTuple, filter_sorted, flatten
from devito.types import Dimension

__all__ = ['dimension_sort', 'lower_operations']


def dimension_sort(expr):
    """
    Topologically sort the Dimensions in ``expr``, based on the order in which they
    appear within Indexeds.
    """

    def handle_indexed(indexed):
        relation = []
        for i in indexed.indices:
            try:
                maybe_dim = split_affine(i).var
                if isinstance(maybe_dim, Dimension):
                    relation.append(maybe_dim)
            except ValueError:
                # Maybe there are some nested Indexeds (e.g., the situation is A[B[i]])
                nested = flatten(handle_indexed(n) for n in retrieve_indexed(i))
                if nested:
                    relation.extend(nested)
                else:
                    # Fallback: Just insert all the Dimensions we find, regardless of
                    # what the user is attempting to do
                    relation.extend([d for d in filter_sorted(i.free_symbols)
                                     if isinstance(d, Dimension)])
        return tuple(relation)

    relations = {handle_indexed(i) for i in retrieve_indexed(expr)}

    # Add in any implicit dimension (typical of scalar temporaries, or Step)
    relations.add(expr.implicit_dims)

    # Add in leftover free dimensions (not an Indexed' index)
    extra = set([i for i in expr.free_symbols if isinstance(i, Dimension)])

    # Add in pure data dimensions (e.g., those accessed only via explicit values,
    # such as A[3])
    indexeds = retrieve_indexed(expr, deep=True)
    extra.update(set().union(*[set(i.function.dimensions) for i in indexeds]))

    # Enforce determinism
    extra = filter_sorted(extra, key=attrgetter('name'))

    # Add in implicit relations for parent dimensions
    # -----------------------------------------------
    # 1) Note that (d.parent, d) is what we want, while (d, d.parent) would be
    # wrong; for example, in `((t, time), (t, x, y), (x, y))`, `x` could now
    # preceed `time`, while `t`, and therefore `time`, *must* appear before `x`,
    # as indicated by the second relation
    implicit_relations = {(d.parent, d) for d in extra if d.is_Derived}
    # 2) To handle cases such as `((time, xi), (x,))`, where `xi` a SubDimension
    # of `x`, besides `(x, xi)`, we also have to add `(time, x)` so that we
    # obtain the desired ordering `(time, x, xi)`. W/o `(time, x)`, the ordering
    # `(x, time, xi)` might be returned instead, which would be non-sense
    implicit_relations.update({tuple(d.root for d in i) for i in relations})

    ordering = PartialOrderTuple(extra, relations=(relations | implicit_relations))

    return ordering


def lower_operations(expr):
    """
    Construct an expression semantically equivalent to ``expr`` in which all
    operations of type Differentiable have been lowered to SymPy operations.
    """

    def _lower_operations(obj):
        flag = False
        args = []
        for a in obj.args:
            ax, af = _lower_operations(a)
            args.append(ax)
            flag |= af
        if isinstance(obj, DifferentiableOp):
            for cls in obj.__class__.mro()[1:]:
                if obj.__class__.__name__ == cls.__name__:
                    return cls(*args, evaluate=False), True
            assert False
        elif flag:
            return obj.func(*args, evaluate=False), True
        else:
            return obj, False

    return _lower_operations(expr)[0]
