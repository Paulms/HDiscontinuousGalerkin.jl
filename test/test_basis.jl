using HDiscontinuousGalerkin

const rr = 0.5
const ss = 0.5
interpolation = Dubiner{2,RefSimplex,4}()
for j = 1:15
    @test dubiner_basis(rr,ss,j) ≈ value(interpolation, j, (rr, ss))
    @test ∇dubiner_basis(rr,ss,j) ≈ gradient_value(interpolation, j, (rr, ss))
end
