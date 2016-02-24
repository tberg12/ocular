package edu.berkeley.cs.nlp.ocular.util;

import java.io.Serializable;
import java.util.Comparator;

/**
 * @author Dan Klein
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class Tuple3<A1, A2, A3> implements Serializable {
	static final long serialVersionUID = 53;

	public final A1 _1;
	public final A2 _2;
	public final A3 _3;

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof Tuple3))
			return false;

		@SuppressWarnings("rawtypes")
		final Tuple3 tuple = (Tuple3) o;

		if (_1 != null ? !_1.equals(tuple._1) : tuple._1 != null)
			return false;
		if (_2 != null ? !_2.equals(tuple._2) : tuple._2 != null)
			return false;
		if (_3 != null ? !_3.equals(tuple._3) : tuple._3 != null)
			return false;

		return true;
	}

	public int hashCode() {
		int result;
		result = (_1 != null ? _1.hashCode() : 0);
		result = 29 * result + (_2 != null ? _2.hashCode() : 0);
		result = 31 * result + (_3 != null ? _3.hashCode() : 0);
		return result;
	}

	public String toString() {
		return "(" + _1 + ", " + _2 + ", " + _3 + ")";
	}

	public Tuple3(A1 _1, A2 _2, A3 _3) {
		this._1 = _1;
		this._2 = _2;
		this._3 = _3;
	}

	public static <A1, A2, A3> Tuple3<A1, A2, A3> Tuple3(A1 _1, A2 _2, A3 _3) {
		return new Tuple3<A1, A2, A3>(_1, _2, _3);
	}

	public static class LexicographicTuple3Comparator<A1, A2, A3> implements Comparator<Tuple3<A1, A2, A3>> {
		Comparator<A1> _1Comparator;
		Comparator<A2> _2Comparator;
		Comparator<A3> _3Comparator;

		public int compare(Tuple3<A1, A2, A3> tuple1, Tuple3<A1, A2, A3> tuple2) {
			int _1Compare = _1Comparator.compare(tuple1._1, tuple2._1);
			if (_1Compare != 0)
				return _1Compare;
			int _2Compare = _2Comparator.compare(tuple1._2, tuple2._2);
			if (_2Compare != 0)
				return _2Compare;
			return _3Comparator.compare(tuple1._3, tuple2._3);
		}

		public LexicographicTuple3Comparator(Comparator<A1> _1Comparator, Comparator<A2> _2Comparator, Comparator<A3> _3Comparator) {
			this._1Comparator = _1Comparator;
			this._2Comparator = _2Comparator;
			this._3Comparator = _3Comparator;
		}
	}

	public static class DefaultLexicographicTuple3Comparator<A1 extends Comparable<A1>, A2 extends Comparable<A2>, A3 extends Comparable<A3>>
			implements Comparator<Tuple3<A1, A2, A3>> {

		public int compare(Tuple3<A1, A2, A3> x, Tuple3<A1, A2, A3> y) {
			int _1Compare = x._1.compareTo(y._1);
			if (_1Compare != 0) {
				return _1Compare;
			}
			int _2Compare = x._2.compareTo(y._2);
			if (_2Compare != 0) {
				return _2Compare;
			}
			return x._3.compareTo(y._3);
		}

	}

}
