import NullDashboard from "./NullDiscovery";
import DataValidationDashboard from "./TypeValidation";

const Validation = ({ data }) => {
    return (
        <div>
            <NullDashboard data={data.null_analysis} />

            {data?.col_analysis && <DataValidationDashboard columnData={data.col_analysis} />}
        </div>
    );
}
export default Validation;