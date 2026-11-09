"""Assignment 3 H2O AutoML requirements 2–5.
Runs locally; no dataset upload to an external service.
"""
import argparse, hashlib, json, os, platform, time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.runtime'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import h2o
from h2o.automl import H2OAutoML


def md_table(df):
    headers=list(df.columns)
    def fmt(v):
        return f'{v:.4f}' if isinstance(v,float) else str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'athletes.csv')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'h2o_results')
    parser.add_argument('--max-models',type=int,default=10,help='Base model limit per feature set, excluding ensembles')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'cluster_tmp').mkdir(exist_ok=True)
    # Use the task-local Java runtime if installed, without changing OS configuration.
    runtime=root/'.runtime'
    java_homes=list(runtime.glob('*/Contents/Home')) if runtime.exists() else []
    if java_homes:os.environ['JAVA_HOME']=str(java_homes[0])
    df=pd.read_csv(args.data)
    target='deadlift'
    if target not in df or not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError('Expected numeric deadlift target')
    if df[target].isna().any():raise ValueError('Missing targets require an explicit cleaning decision')
    predictors=[c for c in df if c!=target]
    df.describe(include='all').to_csv(out/'data_summary.csv')
    df.isna().sum().rename('missing').to_csv(out/'missing_counts.csv')
    df.nunique().rename('unique').to_csv(out/'unique_counts.csv')
    metadata={'source_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'rows':len(df),'columns':list(df),'target':target,'seed':42,'max_base_models_per_run':args.max_models,'platform':platform.platform(),'python_version':platform.python_version(),'h2o_version':h2o.__version__,'algorithm_scope':['GLM','GBM','DRF','StackedEnsemble'],'categorical_columns':list(df.select_dtypes(include=['object','string']).columns)}
    # Bind to loopback. Do not attach to or modify a pre-existing H2O cluster.
    h2o.init(ip='127.0.0.1',port=54331,bind_to_localhost=True,start_h2o=True,nthreads=2,max_mem_size='4G',ice_root=str(out/'cluster_tmp'))
    h2o.no_progress()
    try:
        frame=h2o.H2OFrame(df)
        for col in metadata['categorical_columns']:frame[col]=frame[col].asfactor()
        # Fixed assignment exported for reproducibility and reused for both feature sets.
        assignment=frame.runif(seed=42)
        train=frame[assignment<.64];valid=frame[(assignment>=.64)&(assignment<.80)];test=frame[assignment>=.80]
        splits=assignment.as_data_frame().iloc[:,0].map(lambda x:'train' if x<.64 else ('validation' if x<.80 else 'test'))
        pd.DataFrame({'source_row_zero_based':range(len(df)),'partition':splits}).to_csv(out/'partition_assignment.csv',index=False)
        metadata['partition_rows']={'train':train.nrows,'validation':valid.nrows,'test':test.nrows}
        run_tag=time.strftime('%Y%m%d_%H%M%S')
        def run(label,features):
            print(f'START {label}: {len(features)} predictors',flush=True)
            aml=H2OAutoML(max_models=args.max_models,seed=42,nfolds=5,sort_metric='RMSE',stopping_metric='RMSE',include_algos=metadata['algorithm_scope'],project_name=f'athletes_{label}_{run_tag}',verbosity='info')
            start=time.perf_counter()
            # CV controls early stopping. The validation frame ranks models only.
            aml.train(x=features,y=target,training_frame=train,leaderboard_frame=valid)
            elapsed=time.perf_counter()-start
            lb=aml.leaderboard.as_data_frame().sort_values('rmse').reset_index(drop=True)
            lb.to_csv(out/f'{label}_leaderboard.csv',index=False)
            lb.head(3).to_csv(out/f'{label}_top3.csv',index=False)
            aml.event_log.as_data_frame().to_csv(out/f'{label}_event_log.csv',index=False)
            # Export a model manifest with algorithm and CV RMSE for interpretation.
            manifest=[]
            for mid in lb.model_id:
                m=h2o.get_model(mid)
                manifest.append({'model_id':mid,'algorithm':m.algo,'cv_rmse':m.rmse(xval=True),'validation_rmse':float(lb.loc[lb.model_id==mid,'rmse'].iloc[0])})
            pd.DataFrame(manifest).to_csv(out/f'{label}_model_manifest.csv',index=False)
            metadata[label]={'predictors':features,'elapsed_seconds':elapsed,'model_count':len(lb),'leader_id':aml.leader.model_id,'leader_algorithm':aml.leader.algo,'validation_rmse':float(lb.iloc[0].rmse)}
            print(f'FINISH {label}: {len(lb)} models, best validation RMSE {lb.iloc[0].rmse:.4f}',flush=True)
            return aml,lb
        full,full_lb=run('all_features',predictors)
        # Use native H2O importance from the best validation-ranked eligible model.
        # Ensembles do not expose comparable native variable importance.
        importance=None
        for mid in full_lb.model_id:
            model=h2o.get_model(mid)
            if model.algo=='stackedensemble':continue
            vi=model.varimp(use_pandas=True)
            if vi is not None and len(vi)>=5 and set(vi.variable.head(5))<=set(predictors):
                importance=vi;metadata['importance_model_id']=mid;metadata['importance_model_algorithm']=model.algo;break
        if importance is None:raise RuntimeError('No eligible model has importance for five original predictors')
        importance.to_csv(out/'feature_importance.csv',index=False)
        top5=importance.variable.head(5).tolist();top3=top5[:3]
        metadata['top_five_features']=top5;metadata['top_three_features']=top3
        print('TOP FIVE FEATURES: '+', '.join(top5),flush=True)
        reduced,reduced_lb=run('top3_features',top3)
        # Evaluate held-out test only after final selection within each feature set.
        for label,aml in [('all_features',full),('top3_features',reduced)]:
            perf=aml.leader.model_performance(test_data=test)
            metadata[label]['test_rmse']=perf.rmse();metadata[label]['test_mae']=perf.mae();metadata[label]['test_r2']=perf.r2()
            model_dir=out/'models';model_dir.mkdir(exist_ok=True)
            metadata[label]['model_file']=h2o.save_model(aml.leader,path=str(model_dir),force=False)
            predictions=aml.leader.predict(test).as_data_frame().iloc[:,0]
            actual=test[target].as_data_frame().iloc[:,0]
            pd.DataFrame({'actual':actual,'predicted':predictions,'residual':actual-predictions}).to_csv(out/f'{label}_test_predictions.csv',index=False)
        (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
        fig,ax=plt.subplots(figsize=(8,4.4));v=importance.head(5).iloc[::-1]
        ax.barh(v.variable,v.relative_importance,color='#416c9b');ax.set_xlabel('H2O relative importance');ax.set_title('Top five features from '+metadata['importance_model_algorithm']);fig.tight_layout();fig.savefig(out/'top_five_features.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.4));ax.hist(df[target],bins=50,color='#416c9b');ax.set(xlabel='Deadlift',ylabel='Athletes',title='Target distribution');fig.tight_layout();fig.savefig(out/'target_distribution.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.4))
        for label,lb in [('All features',full_lb),('Top three features',reduced_lb)]:ax.plot(range(1,min(10,len(lb))+1),lb.rmse.head(10),marker='o',label=label)
        ax.set(xlabel='Rank within feature set',ylabel='Validation RMSE',title='H2O AutoML validation rankings');ax.legend();fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=180);plt.close(fig)
        cols=['model_id','rmse','mae','rmsle']
        def leaders(lb):
            x=lb.head(3)[[c for c in cols if c in lb]].copy();x.insert(0,'Rank',[1,2,3]);return md_table(x)
        stats=df[target].describe();missing=int(df.isna().sum().sum())
        numeric=df.select_dtypes(include='number').corr()[target].drop(target).sort_values(key=abs,ascending=False)
        full_best=metadata['all_features'];red_best=metadata['top3_features']
        report=f'''# Assignment 3 H2O AutoML Analysis

## 2 Find the best model using AutoML

I used H2O AutoML to predict **deadlift** from the supplied athletes dataset. The dataset contains **{len(df):,} rows**, **{len(predictors)} predictors**, and one target. Both searches used identical random partitions (seed 42): {train.nrows:,} training rows, {valid.nrows:,} validation rows, and {test.nrows:,} test rows. Five-fold cross-validation was used within the training partition. The validation partition ranked candidate models; the test partition was evaluated only after selection.

Each search allowed {args.max_models} base models plus stacked ensembles. The search included GLM, GBM, distributed random forests, and stacked ensembles. XGBoost and Deep Learning were not included in this local search. String columns were encoded as categorical variables, including the survey response fields. This differs from DataRobot's treatment of some fields as text.

The best all-feature model was **{full_best['leader_id']}** ({full_best['leader_algorithm']}), with validation RMSE **{full_best['validation_rmse']:.4f}**. Its final test RMSE was **{full_best['test_rmse']:.4f}**. “Best” refers to the models evaluated within this configured search. Lower RMSE is better.

## 3 Data insights

- The dataset contains **{missing:,} missing cells** and **{int(df.duplicated().sum()):,} exact duplicate row(s)**. No records were removed or values changed.
- Deadlift has mean **{stats['mean']:.2f}**, median **{stats['50%']:.2f}**, and standard deviation **{stats['std']:.2f}**, ranging from **{stats['min']:.0f}** to **{stats['max']:.0f}**.
- The strongest numeric Pearson correlation with deadlift is **{numeric.index[0]}** at **{numeric.iloc[0]:.3f}**. Correlation and feature importance describe association, not causation.
- Very low lift values and extreme body measurements merit review; this analysis retains the uploaded data rather than assuming those values are errors.
- The best three-feature model has validation RMSE **{red_best['validation_rmse']:.4f}**, compared with **{full_best['validation_rmse']:.4f}** using all features. The feature sets use the same validation rows.

![Target distribution](h2o_results/target_distribution.png)

## 4 Top five reported features

H2O native variable importance is reported from **{metadata['importance_model_id']}** ({metadata['importance_model_algorithm']}), the highest-validation-ranked model that supports importance for original columns. This is explicitly the importance of that model; it is not an ensemble importance score if the overall leader is an ensemble. The ranking is derived from model training, not from test data.

{md_table(importance.head(5))}

The `percentage` column above is a fraction of total native importance (for example, 0.7020 means 70.20%).

The reduced search uses **{', '.join(top3)}**.

![Top five features](h2o_results/top_five_features.png)

## 5 Top three models by validation score

### All features

{leaders(full_lb)}

### Only the top three features

{leaders(reduced_lb)}

![Validation comparison](h2o_results/validation_comparison.png)

## Reproducibility and limitations

Run `./run_h2o.sh` from the assignment folder. Code is in `h2o_automl.py`; dependencies are pinned in `requirements_h2o.txt`. Full leaderboards, model manifests, event logs, feature importance, data summaries, exported partition assignments, metadata, test predictions, and fitted leader models are saved in `h2o_results/`. H2O version: {h2o.__version__}. The input SHA256 is `{metadata['source_sha256']}`.

The H2O split is reproducible but does not reproduce DataRobot's exact partition membership, preprocessing, or search budget. Cross-platform score differences should therefore be interpreted descriptively. H2O is run through Python code, so screenshots are not required for this code-based workflow. The Assignment 1 Part A comparison remains deferred.

Reference: [H2O AutoML documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html). A supplied `leaderboard_frame` determines leaderboard scores; with five-fold CV, early stopping uses CV rather than a separate validation frame.
'''
        (root/'H2O_Analysis.md').write_text(report)
        print('REPORT SAVED: '+str(root/'H2O_Analysis.md'),flush=True)
    finally:
        h2o.cluster().shutdown(prompt=False)

if __name__=='__main__':main()
