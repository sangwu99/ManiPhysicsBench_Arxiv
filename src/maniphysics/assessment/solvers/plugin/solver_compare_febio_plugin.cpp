

#include <FECore/sdk.h>
#include <FECore/version.h>
#include <FECore/FELogElemData.h>
#include <FECore/FEElement.h>
#include <FEBioMech/FEElasticMaterialPoint.h>
#include <FEBioMech/FEDiscreteElasticMaterial.h>
#include <FEBioMech/FEElasticMaterial.h>
#include <FECore/FESolidDomain.h>
#include <FEBioMech/FESSIShellDomain.h>
#include <FEBioMech/FEVonMisesPlasticity.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

template<int mode> class IPDiagnostic : public FELogElemData {
public:
    IPDiagnostic(FEModel* m) : FELogElemData(m) {}
    double value(FEElement& el) override {
        double val = -std::numeric_limits<double>::infinity();
        if (mode == 3) return el.GaussPoints();
        for (int i=0; i<el.GaussPoints(); ++i) {
            auto* p=el.GetMaterialPoint(i)->ExtractData<FEElasticMaterialPoint>();
            if (!p) throw std::runtime_error("IP diagnostic requires elastic material points");
            double x;
            if (mode == 0) x=p->m_s.effective_norm();
            else if (mode == 1) {
                double eig[3]; p->m_s.eigen(eig); x=*std::max_element(eig,eig+3);
            } else x=-p->m_J;
            val=std::max(val,x);
        }
        return mode == 2 ? -val : val;
    }
};
using IPVM=IPDiagnostic<0>;
using IPS1=IPDiagnostic<1>;
using MinJ=IPDiagnostic<2>;
using IPCount=IPDiagnostic<3>;

template<bool kinetic> class ElementEnergy : public FELogElemData {
public:
    ElementEnergy(FEModel* m) : FELogElemData(m) {}
    double value(FEElement& e) override {
        auto* solid=dynamic_cast<FESolidDomain*>(e.GetMeshPartition());
        auto* shell=dynamic_cast<FESSIShellDomain*>(e.GetMeshPartition());
        if (!solid && !shell) throw std::runtime_error("Unsupported energy diagnostic domain");
        auto* material=solid ? solid->GetMaterial() : shell->GetMaterial();
        auto* mat=material->ExtractProperty<FEElasticMaterial>();
        if (!mat) throw std::runtime_error("Energy diagnostic requires elastic material");
        double sum=0;
        for (int n=0;n<e.GaussPoints();++n) {
            auto& mp=*e.GetMaterialPoint(n);
            auto* p=mp.ExtractData<FEElasticMaterialPoint>();
            double w=kinetic ? .5*mat->Density(mp)*(p->m_v*p->m_v) : mat->StrainEnergyDensity(mp);
            double jac=solid ? solid->detJ0(dynamic_cast<FESolidElement&>(e),n)
                             : shell->detJ0(dynamic_cast<FEShellElement&>(e),n);
            double weight=solid ? dynamic_cast<FESolidElement&>(e).GaussWeights()[n]
                                : dynamic_cast<FEShellElement&>(e).GaussWeights()[n];
            sum+=w*jac*weight;
        }
        return sum;
    }
};
using ElementSE=ElementEnergy<false>;
using ElementKE=ElementEnergy<true>;

class CartesianTether : public FEDiscreteElasticMaterial {
public:
    CartesianTether(FEModel* m) : FEDiscreteElasticMaterial(m) {}
    double k=0;
    vec3d Force(FEDiscreteMaterialPoint& p) override {return (p.m_drt-p.m_dr0)*k;}
    mat3d Stiffness(FEDiscreteMaterialPoint&) override {
        mat3d a; a.zero(); a[0][0]=a[1][1]=a[2][2]=k; return a;
    }
    double StrainEnergy(FEDiscreteMaterialPoint& p) override {
        auto u=p.m_drt-p.m_dr0; return .5*k*(u*u);
    }
    DECLARE_FECORE_CLASS();
};
BEGIN_FECORE_CLASS(CartesianTether, FEDiscreteElasticMaterial)
    ADD_PARAMETER(k, FE_RANGE_GREATER(0.0), "k");
END_FECORE_CLASS();



class TrackedJ2Point : public FEJ2PlasticMaterialPoint {
public:
    double q0=0, q1=0;
    FEMaterialPointData* Copy() override {return new TrackedJ2Point(*this);}
    void Init() override {FEJ2PlasticMaterialPoint::Init();q0=q1=0;}
    void Update(const FETimeInfo& t) override {q0=q1;FEJ2PlasticMaterialPoint::Update(t);}
    void Serialize(DumpStream& ar) override {FEJ2PlasticMaterialPoint::Serialize(ar);ar & q0 & q1;}
};
class TrackedJ2 : public FEVonMisesPlasticity {
public:
    TrackedJ2(FEModel* m):FEVonMisesPlasticity(m){}
    FEMaterialPointData* CreateMaterialPointData() override {
        auto* p=new TrackedJ2Point;p->Y0=m_Y;return p;
    }
    mat3ds Stress(FEMaterialPoint& mp) override {
        auto* p=mp.ExtractData<TrackedJ2Point>();
        mat3ds de=p->m_F.sym()-mat3dd(1.)-p->e0;
        mat3ds trial=p->sn+de.dev()*(2*m_G)+de.iso()*(3*m_K);
        double L=std::max(0.,(trial.dev().norm()-sqrt(2./3.)*p->Y0)/(2*m_G+m_H));
        p->q1=p->q0+sqrt(2./3.)*L;
        return FEVonMisesPlasticity::Stress(mp);
    }
    DECLARE_FECORE_CLASS();
};
BEGIN_FECORE_CLASS(TrackedJ2, FEVonMisesPlasticity)
END_FECORE_CLASS();
class MeanPEEQ : public FELogElemData {
public:
    MeanPEEQ(FEModel* m):FELogElemData(m){}
    double value(FEElement& e) override {
        double sum=0;
        for(int i=0;i<e.GaussPoints();++i) {
            auto* p=e.GetMaterialPoint(i)->ExtractData<TrackedJ2Point>();
            if(!p)throw std::runtime_error("PEEQ requires tracked native J2 material");
            sum+=p->q1;
        }
        return sum/e.GaussPoints();
    }
};


template<int ip,int component> class StressIP : public FELogElemData {
public:
    StressIP(FEModel* m):FELogElemData(m){}
    double value(FEElement& e) override {
        if(ip>=e.GaussPoints())throw std::runtime_error("Missing requested integration point");
        auto* p=e.GetMaterialPoint(ip)->ExtractData<FEElasticMaterialPoint>();
        const auto& s=p->m_s;
        const double a[]={s.xx(),s.yy(),s.zz(),s.xy(),s.yz(),s.xz()};
        return a[component];
    }
};

using Stress0_0=StressIP<0,0>;
using Stress0_1=StressIP<0,1>;
using Stress0_2=StressIP<0,2>;
using Stress0_3=StressIP<0,3>;
using Stress0_4=StressIP<0,4>;
using Stress0_5=StressIP<0,5>;
using Stress1_0=StressIP<1,0>;
using Stress1_1=StressIP<1,1>;
using Stress1_2=StressIP<1,2>;
using Stress1_3=StressIP<1,3>;
using Stress1_4=StressIP<1,4>;
using Stress1_5=StressIP<1,5>;
using Stress2_0=StressIP<2,0>;
using Stress2_1=StressIP<2,1>;
using Stress2_2=StressIP<2,2>;
using Stress2_3=StressIP<2,3>;
using Stress2_4=StressIP<2,4>;
using Stress2_5=StressIP<2,5>;
using Stress3_0=StressIP<3,0>;
using Stress3_1=StressIP<3,1>;
using Stress3_2=StressIP<3,2>;
using Stress3_3=StressIP<3,3>;
using Stress3_4=StressIP<3,4>;
using Stress3_5=StressIP<3,5>;
using Stress4_0=StressIP<4,0>;
using Stress4_1=StressIP<4,1>;
using Stress4_2=StressIP<4,2>;
using Stress4_3=StressIP<4,3>;
using Stress4_4=StressIP<4,4>;
using Stress4_5=StressIP<4,5>;
using Stress5_0=StressIP<5,0>;
using Stress5_1=StressIP<5,1>;
using Stress5_2=StressIP<5,2>;
using Stress5_3=StressIP<5,3>;
using Stress5_4=StressIP<5,4>;
using Stress5_5=StressIP<5,5>;
using Stress6_0=StressIP<6,0>;
using Stress6_1=StressIP<6,1>;
using Stress6_2=StressIP<6,2>;
using Stress6_3=StressIP<6,3>;
using Stress6_4=StressIP<6,4>;
using Stress6_5=StressIP<6,5>;
using Stress7_0=StressIP<7,0>;
using Stress7_1=StressIP<7,1>;
using Stress7_2=StressIP<7,2>;
using Stress7_3=StressIP<7,3>;
using Stress7_4=StressIP<7,4>;
using Stress7_5=StressIP<7,5>;
using Stress8_0=StressIP<8,0>;
using Stress8_1=StressIP<8,1>;
using Stress8_2=StressIP<8,2>;
using Stress8_3=StressIP<8,3>;
using Stress8_4=StressIP<8,4>;
using Stress8_5=StressIP<8,5>;
using Stress9_0=StressIP<9,0>;
using Stress9_1=StressIP<9,1>;
using Stress9_2=StressIP<9,2>;
using Stress9_3=StressIP<9,3>;
using Stress9_4=StressIP<9,4>;
using Stress9_5=StressIP<9,5>;
using Stress10_0=StressIP<10,0>;
using Stress10_1=StressIP<10,1>;
using Stress10_2=StressIP<10,2>;
using Stress10_3=StressIP<10,3>;
using Stress10_4=StressIP<10,4>;
using Stress10_5=StressIP<10,5>;
using Stress11_0=StressIP<11,0>;
using Stress11_1=StressIP<11,1>;
using Stress11_2=StressIP<11,2>;
using Stress11_3=StressIP<11,3>;
using Stress11_4=StressIP<11,4>;
using Stress11_5=StressIP<11,5>;
using Stress12_0=StressIP<12,0>;
using Stress12_1=StressIP<12,1>;
using Stress12_2=StressIP<12,2>;
using Stress12_3=StressIP<12,3>;
using Stress12_4=StressIP<12,4>;
using Stress12_5=StressIP<12,5>;
using Stress13_0=StressIP<13,0>;
using Stress13_1=StressIP<13,1>;
using Stress13_2=StressIP<13,2>;
using Stress13_3=StressIP<13,3>;
using Stress13_4=StressIP<13,4>;
using Stress13_5=StressIP<13,5>;
using Stress14_0=StressIP<14,0>;
using Stress14_1=StressIP<14,1>;
using Stress14_2=StressIP<14,2>;
using Stress14_3=StressIP<14,3>;
using Stress14_4=StressIP<14,4>;
using Stress14_5=StressIP<14,5>;
using Stress15_0=StressIP<15,0>;
using Stress15_1=StressIP<15,1>;
using Stress15_2=StressIP<15,2>;
using Stress15_3=StressIP<15,3>;
using Stress15_4=StressIP<15,4>;
using Stress15_5=StressIP<15,5>;
using Stress16_0=StressIP<16,0>;
using Stress16_1=StressIP<16,1>;
using Stress16_2=StressIP<16,2>;
using Stress16_3=StressIP<16,3>;
using Stress16_4=StressIP<16,4>;
using Stress16_5=StressIP<16,5>;
using Stress17_0=StressIP<17,0>;
using Stress17_1=StressIP<17,1>;
using Stress17_2=StressIP<17,2>;
using Stress17_3=StressIP<17,3>;
using Stress17_4=StressIP<17,4>;
using Stress17_5=StressIP<17,5>;
using Stress18_0=StressIP<18,0>;
using Stress18_1=StressIP<18,1>;
using Stress18_2=StressIP<18,2>;
using Stress18_3=StressIP<18,3>;
using Stress18_4=StressIP<18,4>;
using Stress18_5=StressIP<18,5>;
using Stress19_0=StressIP<19,0>;
using Stress19_1=StressIP<19,1>;
using Stress19_2=StressIP<19,2>;
using Stress19_3=StressIP<19,3>;
using Stress19_4=StressIP<19,4>;
using Stress19_5=StressIP<19,5>;
using Stress20_0=StressIP<20,0>;
using Stress20_1=StressIP<20,1>;
using Stress20_2=StressIP<20,2>;
using Stress20_3=StressIP<20,3>;
using Stress20_4=StressIP<20,4>;
using Stress20_5=StressIP<20,5>;
FECORE_PLUGIN unsigned int GetSDKVersion() {return FE_SDK_VERSION;}
FECORE_PLUGIN void GetPluginVersion(int& a,int& b,int& c) {a=1;b=1;c=0;}
FECORE_PLUGIN void PluginInitialize(FECoreKernel& kernel) {
    FECoreKernel::SetInstance(&kernel); kernel.SetActiveModule("solid");
    REGISTER_FECORE_CLASS(TrackedJ2,"comparison native tracked J2");
    REGISTER_FECORE_CLASS(MeanPEEQ,"comparison mean peeq");
    REGISTER_FECORE_CLASS(Stress20_5,"comparison s20_5");
    REGISTER_FECORE_CLASS(Stress20_4,"comparison s20_4");
    REGISTER_FECORE_CLASS(Stress20_3,"comparison s20_3");
    REGISTER_FECORE_CLASS(Stress20_2,"comparison s20_2");
    REGISTER_FECORE_CLASS(Stress20_1,"comparison s20_1");
    REGISTER_FECORE_CLASS(Stress20_0,"comparison s20_0");
    REGISTER_FECORE_CLASS(Stress19_5,"comparison s19_5");
    REGISTER_FECORE_CLASS(Stress19_4,"comparison s19_4");
    REGISTER_FECORE_CLASS(Stress19_3,"comparison s19_3");
    REGISTER_FECORE_CLASS(Stress19_2,"comparison s19_2");
    REGISTER_FECORE_CLASS(Stress19_1,"comparison s19_1");
    REGISTER_FECORE_CLASS(Stress19_0,"comparison s19_0");
    REGISTER_FECORE_CLASS(Stress18_5,"comparison s18_5");
    REGISTER_FECORE_CLASS(Stress18_4,"comparison s18_4");
    REGISTER_FECORE_CLASS(Stress18_3,"comparison s18_3");
    REGISTER_FECORE_CLASS(Stress18_2,"comparison s18_2");
    REGISTER_FECORE_CLASS(Stress18_1,"comparison s18_1");
    REGISTER_FECORE_CLASS(Stress18_0,"comparison s18_0");
    REGISTER_FECORE_CLASS(Stress17_5,"comparison s17_5");
    REGISTER_FECORE_CLASS(Stress17_4,"comparison s17_4");
    REGISTER_FECORE_CLASS(Stress17_3,"comparison s17_3");
    REGISTER_FECORE_CLASS(Stress17_2,"comparison s17_2");
    REGISTER_FECORE_CLASS(Stress17_1,"comparison s17_1");
    REGISTER_FECORE_CLASS(Stress17_0,"comparison s17_0");
    REGISTER_FECORE_CLASS(Stress16_5,"comparison s16_5");
    REGISTER_FECORE_CLASS(Stress16_4,"comparison s16_4");
    REGISTER_FECORE_CLASS(Stress16_3,"comparison s16_3");
    REGISTER_FECORE_CLASS(Stress16_2,"comparison s16_2");
    REGISTER_FECORE_CLASS(Stress16_1,"comparison s16_1");
    REGISTER_FECORE_CLASS(Stress16_0,"comparison s16_0");
    REGISTER_FECORE_CLASS(Stress15_5,"comparison s15_5");
    REGISTER_FECORE_CLASS(Stress15_4,"comparison s15_4");
    REGISTER_FECORE_CLASS(Stress15_3,"comparison s15_3");
    REGISTER_FECORE_CLASS(Stress15_2,"comparison s15_2");
    REGISTER_FECORE_CLASS(Stress15_1,"comparison s15_1");
    REGISTER_FECORE_CLASS(Stress15_0,"comparison s15_0");
    REGISTER_FECORE_CLASS(Stress14_5,"comparison s14_5");
    REGISTER_FECORE_CLASS(Stress14_4,"comparison s14_4");
    REGISTER_FECORE_CLASS(Stress14_3,"comparison s14_3");
    REGISTER_FECORE_CLASS(Stress14_2,"comparison s14_2");
    REGISTER_FECORE_CLASS(Stress14_1,"comparison s14_1");
    REGISTER_FECORE_CLASS(Stress14_0,"comparison s14_0");
    REGISTER_FECORE_CLASS(Stress13_5,"comparison s13_5");
    REGISTER_FECORE_CLASS(Stress13_4,"comparison s13_4");
    REGISTER_FECORE_CLASS(Stress13_3,"comparison s13_3");
    REGISTER_FECORE_CLASS(Stress13_2,"comparison s13_2");
    REGISTER_FECORE_CLASS(Stress13_1,"comparison s13_1");
    REGISTER_FECORE_CLASS(Stress13_0,"comparison s13_0");
    REGISTER_FECORE_CLASS(Stress12_5,"comparison s12_5");
    REGISTER_FECORE_CLASS(Stress12_4,"comparison s12_4");
    REGISTER_FECORE_CLASS(Stress12_3,"comparison s12_3");
    REGISTER_FECORE_CLASS(Stress12_2,"comparison s12_2");
    REGISTER_FECORE_CLASS(Stress12_1,"comparison s12_1");
    REGISTER_FECORE_CLASS(Stress12_0,"comparison s12_0");
    REGISTER_FECORE_CLASS(Stress11_5,"comparison s11_5");
    REGISTER_FECORE_CLASS(Stress11_4,"comparison s11_4");
    REGISTER_FECORE_CLASS(Stress11_3,"comparison s11_3");
    REGISTER_FECORE_CLASS(Stress11_2,"comparison s11_2");
    REGISTER_FECORE_CLASS(Stress11_1,"comparison s11_1");
    REGISTER_FECORE_CLASS(Stress11_0,"comparison s11_0");
    REGISTER_FECORE_CLASS(Stress10_5,"comparison s10_5");
    REGISTER_FECORE_CLASS(Stress10_4,"comparison s10_4");
    REGISTER_FECORE_CLASS(Stress10_3,"comparison s10_3");
    REGISTER_FECORE_CLASS(Stress10_2,"comparison s10_2");
    REGISTER_FECORE_CLASS(Stress10_1,"comparison s10_1");
    REGISTER_FECORE_CLASS(Stress10_0,"comparison s10_0");
    REGISTER_FECORE_CLASS(Stress9_5,"comparison s9_5");
    REGISTER_FECORE_CLASS(Stress9_4,"comparison s9_4");
    REGISTER_FECORE_CLASS(Stress9_3,"comparison s9_3");
    REGISTER_FECORE_CLASS(Stress9_2,"comparison s9_2");
    REGISTER_FECORE_CLASS(Stress9_1,"comparison s9_1");
    REGISTER_FECORE_CLASS(Stress9_0,"comparison s9_0");
    REGISTER_FECORE_CLASS(Stress8_5,"comparison s8_5");
    REGISTER_FECORE_CLASS(Stress8_4,"comparison s8_4");
    REGISTER_FECORE_CLASS(Stress8_3,"comparison s8_3");
    REGISTER_FECORE_CLASS(Stress8_2,"comparison s8_2");
    REGISTER_FECORE_CLASS(Stress8_1,"comparison s8_1");
    REGISTER_FECORE_CLASS(Stress8_0,"comparison s8_0");
    REGISTER_FECORE_CLASS(Stress7_5,"comparison s7_5");
    REGISTER_FECORE_CLASS(Stress7_4,"comparison s7_4");
    REGISTER_FECORE_CLASS(Stress7_3,"comparison s7_3");
    REGISTER_FECORE_CLASS(Stress7_2,"comparison s7_2");
    REGISTER_FECORE_CLASS(Stress7_1,"comparison s7_1");
    REGISTER_FECORE_CLASS(Stress7_0,"comparison s7_0");
    REGISTER_FECORE_CLASS(Stress6_5,"comparison s6_5");
    REGISTER_FECORE_CLASS(Stress6_4,"comparison s6_4");
    REGISTER_FECORE_CLASS(Stress6_3,"comparison s6_3");
    REGISTER_FECORE_CLASS(Stress6_2,"comparison s6_2");
    REGISTER_FECORE_CLASS(Stress6_1,"comparison s6_1");
    REGISTER_FECORE_CLASS(Stress6_0,"comparison s6_0");
    REGISTER_FECORE_CLASS(Stress5_5,"comparison s5_5");
    REGISTER_FECORE_CLASS(Stress5_4,"comparison s5_4");
    REGISTER_FECORE_CLASS(Stress5_3,"comparison s5_3");
    REGISTER_FECORE_CLASS(Stress5_2,"comparison s5_2");
    REGISTER_FECORE_CLASS(Stress5_1,"comparison s5_1");
    REGISTER_FECORE_CLASS(Stress5_0,"comparison s5_0");
    REGISTER_FECORE_CLASS(Stress4_5,"comparison s4_5");
    REGISTER_FECORE_CLASS(Stress4_4,"comparison s4_4");
    REGISTER_FECORE_CLASS(Stress4_3,"comparison s4_3");
    REGISTER_FECORE_CLASS(Stress4_2,"comparison s4_2");
    REGISTER_FECORE_CLASS(Stress4_1,"comparison s4_1");
    REGISTER_FECORE_CLASS(Stress4_0,"comparison s4_0");
    REGISTER_FECORE_CLASS(Stress3_5,"comparison s3_5");
    REGISTER_FECORE_CLASS(Stress3_4,"comparison s3_4");
    REGISTER_FECORE_CLASS(Stress3_3,"comparison s3_3");
    REGISTER_FECORE_CLASS(Stress3_2,"comparison s3_2");
    REGISTER_FECORE_CLASS(Stress3_1,"comparison s3_1");
    REGISTER_FECORE_CLASS(Stress3_0,"comparison s3_0");
    REGISTER_FECORE_CLASS(Stress2_5,"comparison s2_5");
    REGISTER_FECORE_CLASS(Stress2_4,"comparison s2_4");
    REGISTER_FECORE_CLASS(Stress2_3,"comparison s2_3");
    REGISTER_FECORE_CLASS(Stress2_2,"comparison s2_2");
    REGISTER_FECORE_CLASS(Stress2_1,"comparison s2_1");
    REGISTER_FECORE_CLASS(Stress2_0,"comparison s2_0");
    REGISTER_FECORE_CLASS(Stress1_5,"comparison s1_5");
    REGISTER_FECORE_CLASS(Stress1_4,"comparison s1_4");
    REGISTER_FECORE_CLASS(Stress1_3,"comparison s1_3");
    REGISTER_FECORE_CLASS(Stress1_2,"comparison s1_2");
    REGISTER_FECORE_CLASS(Stress1_1,"comparison s1_1");
    REGISTER_FECORE_CLASS(Stress1_0,"comparison s1_0");
    REGISTER_FECORE_CLASS(Stress0_5,"comparison s0_5");
    REGISTER_FECORE_CLASS(Stress0_4,"comparison s0_4");
    REGISTER_FECORE_CLASS(Stress0_3,"comparison s0_3");
    REGISTER_FECORE_CLASS(Stress0_2,"comparison s0_2");
    REGISTER_FECORE_CLASS(Stress0_1,"comparison s0_1");
    REGISTER_FECORE_CLASS(Stress0_0,"comparison s0_0");
    REGISTER_FECORE_CLASS(IPVM,"basis ip max vm");
    REGISTER_FECORE_CLASS(IPS1,"basis ip max s1");
    REGISTER_FECORE_CLASS(MinJ,"basis ip min J");
    REGISTER_FECORE_CLASS(IPCount,"basis ip count");
    REGISTER_FECORE_CLASS(CartesianTether,"basis Cartesian tether");
    REGISTER_FECORE_CLASS(ElementSE,"basis element strain energy");
    REGISTER_FECORE_CLASS(ElementKE,"basis element kinetic energy");
}
