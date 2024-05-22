#[cfg(test)]
mod tests{
    #[test]
fn test_point_add(){
  use std::ops::{Add, Mul};
  use pasta_curves::{arithmetic::CurveExt, group::{cofactor::CofactorCurveAffine, ff::PrimeField}, pallas};
  //point1
  let g = pallas::Affine::generator();
  let s1 = pallas::Scalar::from_u128(123);
  let p1 = pallas::Affine::mul(g, s1);
  let (x1,y1,z1) = p1.jacobian_coordinates();

  //point2
  let s2 = pallas::Scalar::from_u128(456);
  let p2 = pallas::Affine::mul(g, s2);
  let (x2,y2,z2) = p2.jacobian_coordinates();

  //calculate point3 with p1 and p2 jacobian_coordinates
  let a = x1 * z2.square();
  let b = x2 * z1.square() - a;
  let c = y1 * z2.square() * z2;
  let d = y2 * z1.square() * z1 - c;
  let z3 = z1 * z2 * b;
  let x3 = d.square() - b.square() * (b + a.double());
  let y3 = d * (a * b.square() - x3) - c * b.square() * b;

  //point3 == point1 + point2
  let p3 = pallas::Point::new_jacobian(x3, y3, z3).unwrap();
  let p4 = pallas::Point::add(p1, p2);
  let result = pallas::Point::eq(&p3, &p4);
  assert_eq!(result,true);
}
}