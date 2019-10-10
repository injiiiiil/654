package org.pytorch;

import java.util.Locale;
import java.util.Map;

/**
 * Java representation of a TorchScript value, which is implemented as tagged union that can be
 * one of the supported types: https://pytorch.org/docs/stable/jit.html#types .
 * <p>
 * Calling {@code toX} methods for inappropriate types will throw {@link IllegalStateException}.
 * <p>
 * {@code IValue} objects are constructed with {@code IValue.from(value)},
 * {@code IValue.tupleFrom(value1, value2, ...)}, {@code IValue.listFrom(value1, value2, ...)},
 * or one of the {@code dict} methods, depending on the key type.
 * <p>
 * Data is retrieved from {@code IValue} objects with the {@code toX()} methods.  Note that
 * {@code str}-type IValues must be extracted with {@link #toStr()},
 * rather than {@link #toString()}.
 * <p>
 * {@code IValue} objects may retain references to objects passed into their constructors,
 * and may return references to their internal state from {@code toX()}.
 */
public class IValue {
  private static final int TYPE_CODE_NULL = 1;

  private static final int TYPE_CODE_TENSOR = 2;
  private static final int TYPE_CODE_BOOL = 3;
  private static final int TYPE_CODE_LONG = 4;
  private static final int TYPE_CODE_DOUBLE = 5;
  private static final int TYPE_CODE_STRING = 6;

  private static final int TYPE_CODE_TUPLE = 7;
  private static final int TYPE_CODE_BOOL_LIST = 8;
  private static final int TYPE_CODE_LONG_LIST = 9;
  private static final int TYPE_CODE_DOUBLE_LIST = 10;
  private static final int TYPE_CODE_TENSOR_LIST = 11;
  private static final int TYPE_CODE_LIST = 12;

  private static final int TYPE_CODE_DICT_STRING_KEY = 13;
  private static final int TYPE_CODE_DICT_LONG_KEY = 14;

  private final int mTypeCode;
  private Object mData;

  private IValue(int typeCode) {
    this.mTypeCode = typeCode;
  }

  public boolean isNull() {
    return TYPE_CODE_NULL == this.mTypeCode;
  }

  public boolean isTensor() {
    return TYPE_CODE_TENSOR == this.mTypeCode;
  }

  public boolean isBool() {
    return TYPE_CODE_BOOL == this.mTypeCode;
  }

  public boolean isLong() {
    return TYPE_CODE_LONG == this.mTypeCode;
  }

  public boolean isDouble() {
    return TYPE_CODE_DOUBLE == this.mTypeCode;
  }

  public boolean isString() {
    return TYPE_CODE_STRING == this.mTypeCode;
  }

  public boolean isTuple() {
    return TYPE_CODE_TUPLE == this.mTypeCode;
  }

  public boolean isBoolList() {
    return TYPE_CODE_BOOL_LIST == this.mTypeCode;
  }

  public boolean isLongList() {
    return TYPE_CODE_LONG_LIST == this.mTypeCode;
  }

  public boolean isDoubleList() {
    return TYPE_CODE_DOUBLE_LIST == this.mTypeCode;
  }

  public boolean isTensorList() {
    return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
  }

  public boolean isList() {
    return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
  }

  public boolean isDictStringKey() {
    return TYPE_CODE_DICT_STRING_KEY == this.mTypeCode;
  }

  public boolean isDictLongKey() {
    return TYPE_CODE_DICT_LONG_KEY == this.mTypeCode;
  }

  /**
   * Creates a new {@code IValue} of type {@code Optional} that contains no value.
   */
  public static IValue optionalNull() {
    return new IValue(TYPE_CODE_NULL);
  }

  /**
   * Creates a new {@code IValue} of type {@code Tensor}.
   */
  public static IValue from(Tensor tensor) {
    final IValue iv = new IValue(TYPE_CODE_TENSOR);
    iv.mData = tensor;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code bool}.
   */
  public static IValue from(boolean value) {
    final IValue iv = new IValue(TYPE_CODE_BOOL);
    iv.mData = value;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code int}.
   */
  public static IValue from(long value) {
    final IValue iv = new IValue(TYPE_CODE_LONG);
    iv.mData = value;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code float}.
   */
  public static IValue from(double value) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE);
    iv.mData = value;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code str}.
   */
  public static IValue from(String value) {
    final IValue iv = new IValue(TYPE_CODE_STRING);
    iv.mData = value;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code List[bool]}.
   */
  public static IValue listFrom(boolean... list) {
    final IValue iv = new IValue(TYPE_CODE_BOOL_LIST);
    iv.mData = list;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code List[int]}.
   */
  public static IValue listFrom(long... list) {
    final IValue iv = new IValue(TYPE_CODE_LONG_LIST);
    iv.mData = list;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code List[float]}.
   */
  public static IValue listFrom(double... list) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE_LIST);
    iv.mData = list;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code List[Tensor]}.
   */
  public static IValue listFrom(Tensor... list) {
    final IValue iv = new IValue(TYPE_CODE_TENSOR_LIST);
    iv.mData = list;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code List[T]}.  All elements must have the same type.
   */
  public static IValue listFrom(IValue... array) {
    final int size = array.length;
    if (size > 0) {
      final int typeCode0 = array[0].mTypeCode;
      for (int i = 1; i < size; i++) {
        if (typeCode0 != array[i].mTypeCode) {
          throw new IllegalArgumentException("List must contain items of the same type");
        }
      }
    }

    final IValue iv = new IValue(TYPE_CODE_LIST);
    iv.mData = array;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code Tuple[T0, T1, ...]}.
   */
  public static IValue tupleFrom(IValue... array) {
    final IValue iv = new IValue(TYPE_CODE_TUPLE);
    iv.mData = array;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code Dict[str, V]}.
   */
  public static IValue dictStringKeyFrom(Map<String, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_STRING_KEY);
    iv.mData = map;
    return iv;
  }

  /**
   * Creates a new {@code IValue} of type {@code Dict[int, V]}.
   */
  public static IValue dictLongKeyFrom(Map<Long, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_LONG_KEY);
    iv.mData = map;
    return iv;
  }

  public Tensor toTensor() {
    preconditionType(TYPE_CODE_TENSOR, mTypeCode);
    return (Tensor) mData;
  }

  public boolean toBool() {
    preconditionType(TYPE_CODE_BOOL, mTypeCode);
    return (boolean) mData;
  }

  public long toLong() {
    preconditionType(TYPE_CODE_LONG, mTypeCode);
    return (long) mData;
  }

  public double toDouble() {
    preconditionType(TYPE_CODE_DOUBLE, mTypeCode);
    return (double) mData;
  }

  public String toStr() {
    preconditionType(TYPE_CODE_STRING, mTypeCode);
    return (String) mData;
  }

  public boolean[] toBoolList() {
    preconditionType(TYPE_CODE_BOOL_LIST, mTypeCode);
    return (boolean[]) mData;
  }

  public long[] toLongList() {
    preconditionType(TYPE_CODE_LONG_LIST, mTypeCode);
    return (long[]) mData;
  }

  public double[] toDoubleList() {
    preconditionType(TYPE_CODE_DOUBLE_LIST, mTypeCode);
    return (double[]) mData;
  }

  public Tensor[] toTensorList() {
    preconditionType(TYPE_CODE_TENSOR_LIST, mTypeCode);
    return (Tensor[]) mData;
  }

  public IValue[] toList() {
    preconditionType(TYPE_CODE_LIST, mTypeCode);
    return (IValue[]) mData;
  }

  public IValue[] toTuple() {
    preconditionType(TYPE_CODE_TUPLE, mTypeCode);
    return (IValue[]) mData;
  }

  public Map<String, IValue> toDictStringKey() {
    preconditionType(TYPE_CODE_DICT_STRING_KEY, mTypeCode);
    return (Map<String, IValue>) mData;
  }

  public Map<Long, IValue> toDictLongKey() {
    preconditionType(TYPE_CODE_DICT_LONG_KEY, mTypeCode);
    return (Map<Long, IValue>) mData;
  }

  private void preconditionType(int typeCodeExpected, int typeCode) {
    if (typeCode != typeCodeExpected) {
      throw new IllegalStateException(
          String.format(
              Locale.US, "Expected IValue type %d, actual type %d", typeCodeExpected, typeCode));
    }
  }
}
