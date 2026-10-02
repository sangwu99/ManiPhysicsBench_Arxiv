


#define GetSDKVersion BaseGetSDKVersion
#define PluginInitialize BasePluginInitialize
#include "solver_compare_febio_plugin.cpp"
#undef GetSDKVersion
#undef PluginInitialize
#include <FECore/sdk.h>
#include <FECore/version.h>
#include <FECore/FELogElemData.h>
#include <FECore/FEElement.h>
#include <FEBioMech/FEReactivePlasticity.h>
#include <FEBioMech/FEReactivePlasticityMaterialPoint.h>
#include <FEBioMech/FEElasticMaterialPoint.h>
#include <cmath>
#include <stdexcept>


double equiv(const mat3d& f) {
    double ev[3]; (f.transpose()*f).sym().eigen2(ev);
    double e[3], avg=0;
    for(int j=0;j<3;++j) {
        if(ev[j]<=0) throw std::runtime_error("nonpositive plastic metric");
        e[j]=.5*std::log(ev[j]); avg+=e[j]/3;
    }
    double s=0;for(int j=0;j<3;++j)s+=(e[j]-avg)*(e[j]-avg);
    return std::sqrt(2*s/3);
}

class DiagnosticPoint: public FEReactivePlasticityMaterialPoint {
public:
    DiagnosticPoint(FEMaterialPointData* p, FEElasticMaterial* m):FEReactivePlasticityMaterialPoint(p,m){}
    double q0=0;
    FEMaterialPointData* Copy() override {
        auto* p=new DiagnosticPoint(*this);
        if(m_pNext)p->m_pNext=m_pNext->Copy();
        return p;
    }
    double Current() {
        auto* w=ExtractData<FEPlasticFlowCurveMaterialPoint>();
        double q=q0;
        for(size_t i=0;i<m_Fvsi.size();++i)
            q+=w->m_w[i]*equiv(m_Fvsi[i].inverse()*m_Fusi[i]);
        return q;
    }
    void Init() override {FEReactivePlasticityMaterialPoint::Init();q0=0;}
    
    
    void Update(const FETimeInfo& t) override {q0=Current();FEReactivePlasticityMaterialPoint::Update(t);}
    void Serialize(DumpStream& a) override {FEReactivePlasticityMaterialPoint::Serialize(a);a&q0;}
};

class DiagnosticReactive: public FEReactivePlasticity {
public:
    DiagnosticReactive(FEModel* m):FEReactivePlasticity(m){}
    FEMaterialPointData* CreateMaterialPointData() override {
        auto* ep=m_pBase->CreateMaterialPointData();
        auto* fp=m_pFlow->CreateMaterialPointData();fp->SetNext(ep);
        return new DiagnosticPoint(fp,this);
    }
    DECLARE_FECORE_CLASS();
};
BEGIN_FECORE_CLASS(DiagnosticReactive,FEReactivePlasticity)
END_FECORE_CLASS();

template<int mode> class Diagnostic: public FELogElemData {
public:
    Diagnostic(FEModel* m):FELogElemData(m){}
    double value(FEElement& e) override {
        double sum=0;
        for(int n=0;n<e.GaussPoints();++n) {
            auto* p=e.GetMaterialPoint(n)->ExtractData<FEReactivePlasticityMaterialPoint>();
            auto* w=e.GetMaterialPoint(n)->ExtractData<FEPlasticFlowCurveMaterialPoint>();
            if(!p||!w)throw std::runtime_error("reactive material point required");
            if(mode==0) {
                auto* d=e.GetMaterialPoint(n)->ExtractData<DiagnosticPoint>();
                if(!d)throw std::runtime_error("tracked reactive material point required");
                sum+=d->Current();
            }
            else for(size_t i=0;i<p->m_Fvsi.size();++i) {
                if(mode==1)sum+=w->m_w[i]*equiv(p->m_Fvsi[i].inverse());
                if(mode==2)sum+=w->m_w[i]*p->m_gp[i];
            }
        }
        return sum/e.GaussPoints();
    }
};
using Accum=Diagnostic<0>;using Net=Diagnostic<1>;using Oct=Diagnostic<2>;

class YieldSoftening: public DiagnosticReactive {
public:
 YieldSoftening(FEModel* m):DiagnosticReactive(m){}
 double yield0=1, residual=1, scale=1;
 void SetYield(FEMaterialPoint& pt) {
  if(m_pFlow->InitFlowCurve(pt)) pt.ExtractData<DiagnosticPoint>()->Init();
  auto* d=pt.ExtractData<DiagnosticPoint>();
  auto* w=pt.ExtractData<FEPlasticFlowCurveMaterialPoint>();
  if(w->m_Ky.size()!=1 || w->m_w[0]!=1) throw std::runtime_error("Single fully plastic bond family required");
  w->m_Ky[0]=yield0*(residual+(1-residual)*std::exp(-d->q0/scale));
 }
 mat3ds Stress(FEMaterialPoint& p) override {SetYield(p);return DiagnosticReactive::Stress(p);}
 tens4ds Tangent(FEMaterialPoint& p) override {SetYield(p);return DiagnosticReactive::Tangent(p);}
 double StrainEnergyDensity(FEMaterialPoint& p) override {SetYield(p);return DiagnosticReactive::StrainEnergyDensity(p);}
 void UpdateSpecializedMaterialPoints(FEMaterialPoint& p,const FETimeInfo& t) override {SetYield(p);DiagnosticReactive::UpdateSpecializedMaterialPoints(p,t);}
 DECLARE_FECORE_CLASS();
};
BEGIN_FECORE_CLASS(YieldSoftening,DiagnosticReactive)
 ADD_PARAMETER(yield0,FE_RANGE_GREATER(0.0),"yield0");
 ADD_PARAMETER(residual,FE_RANGE_CLOSED(0.001,1.0),"residual");
 ADD_PARAMETER(scale,FE_RANGE_GREATER(0.0),"softening_scale");
END_FECORE_CLASS();

FECORE_PLUGIN int GetSDKVersion(){return FE_SDK_VERSION;}
FECORE_PLUGIN void PluginInitialize(FECoreKernel& k) {
    BasePluginInitialize(k);
    FECoreKernel::SetInstance(&k);
    REGISTER_FECORE_CLASS(YieldSoftening,"history yield softening");
    REGISTER_FECORE_CLASS(DiagnosticReactive,"tracked native reactive plasticity");
    REGISTER_FECORE_CLASS(Accum,"reactive accumulated equivalent plastic strain");
    REGISTER_FECORE_CLASS(Net,"reactive net equivalent plastic strain");
    REGISTER_FECORE_CLASS(Oct,"reactive weighted native octahedral strain");
}
